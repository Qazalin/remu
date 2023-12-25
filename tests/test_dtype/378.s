
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n26>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_u8 v1, v0, s[2:3]                              // 000000001620: DC420000 01020000
	s_mov_b32 s2, 0xb94c1982                                   // 000000001628: BE8200FF B94C1982
	s_mov_b32 s3, 0x37d75334                                   // 000000001630: BE8300FF 37D75334
	s_waitcnt vmcnt(0)                                         // 000000001638: BF8903F7
	v_cvt_f32_ubyte0_e32 v1, v1                                // 00000000163C: 7E022301
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001640: BF870091
	v_mul_f32_e32 v2, 0x3f22f983, v1                           // 000000001644: 100402FF 3F22F983
	v_rndne_f32_e32 v2, v2                                     // 00000000164C: 7E044702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001650: BF870091
	v_fmamk_f32 v3, v2, 0xbfc90fda, v1                         // 000000001654: 58060302 BFC90FDA
	v_fmac_f32_e32 v3, 0xb3a22168, v2                          // 00000000165C: 560604FF B3A22168
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001664: BF8700A1
	v_fmac_f32_e32 v3, 0xa7c234c4, v2                          // 000000001668: 560604FF A7C234C4
	v_cvt_i32_f32_e32 v2, v2                                   // 000000001670: 7E041102
	v_dual_mul_f32 v4, v3, v3 :: v_dual_and_b32 v7, 1, v2      // 000000001674: C8E40703 04060481
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 00000000167C: BF870111
	v_dual_fmaak_f32 v5, s2, v4, 0x3c0881c4 :: v_dual_lshlrev_b32 v2, 30, v2// 000000001680: C8620802 0502049E 3C0881C4
	v_cmp_eq_u32_e32 vcc_lo, 0, v7                             // 00000000168C: 7C940E80
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)// 000000001690: BF870192
	v_and_b32_e32 v2, 0x80000000, v2                           // 000000001694: 360404FF 80000000
	v_fmaak_f32 v5, v4, v5, 0xbe2aaa9d                         // 00000000169C: 5A0A0B04 BE2AAA9D
	v_fmaak_f32 v6, s3, v4, 0xbab64f3b                         // 0000000016A4: 5A0C0803 BAB64F3B
	s_lshl_b64 s[2:3], s[4:5], 2                               // 0000000016AC: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016B0: BF870119
	s_add_u32 s0, s0, s2                                       // 0000000016B4: 80000200
	v_mul_f32_e32 v5, v4, v5                                   // 0000000016B8: 100A0B04
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000016BC: BF8700A2
	v_fmaak_f32 v6, v4, v6, 0x3d2aabf7                         // 0000000016C0: 5A0C0D04 3D2AABF7
	s_addc_u32 s1, s1, s3                                      // 0000000016C8: 82010301
	v_dual_fmac_f32 v3, v3, v5 :: v_dual_fmaak_f32 v6, v4, v6, 0xbf000004// 0000000016CC: C8020B03 03060D04 BF000004
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016D8: BF870091
	v_fma_f32 v4, v4, v6, 1.0                                  // 0000000016DC: D6130004 03CA0D04
	v_cndmask_b32_e32 v3, v4, v3, vcc_lo                       // 0000000016E4: 02060704
	v_cmp_class_f32_e64 vcc_lo, v1, 0x1f8                      // 0000000016E8: D47E006A 0001FF01 000001F8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016F4: BF870092
	v_xor_b32_e32 v2, v2, v3                                   // 0000000016F8: 3A040702
	v_cndmask_b32_e32 v1, 0x7fc00000, v2, vcc_lo               // 0000000016FC: 020204FF 7FC00000
	global_store_b32 v0, v1, s[0:1]                            // 000000001704: DC6A0000 00000100
	s_nop 0                                                    // 00000000170C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001710: BFB60003
	s_endpgm                                                   // 000000001714: BFB00000
