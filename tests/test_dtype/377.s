
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n25>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s0, s6, s15                                      // 000000001618: 80000F06
	s_addc_u32 s1, s7, s3                                      // 00000000161C: 82010307
	global_load_u8 v1, v0, s[0:1]                              // 000000001620: DC420000 01000000
	s_waitcnt vmcnt(0)                                         // 000000001628: BF8903F7
	v_cvt_f32_ubyte0_e32 v1, v1                                // 00000000162C: 7E022301
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001630: BF870121
	v_div_scale_f32 v2, null, v1, v1, 1.0                      // 000000001634: D6FC7C02 03CA0301
	v_div_scale_f32 v5, vcc_lo, 1.0, v1, 1.0                   // 00000000163C: D6FC6A05 03CA02F2
	v_rcp_f32_e32 v3, v2                                       // 000000001644: 7E065502
	s_waitcnt_depctr 0xfff                                     // 000000001648: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 00000000164C: D6130004 23CA0702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001654: BF870091
	v_fmac_f32_e32 v3, v4, v3                                  // 000000001658: 56060704
	v_mul_f32_e32 v4, v5, v3                                   // 00000000165C: 10080705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001660: BF870091
	v_fma_f32 v6, -v2, v4, v5                                  // 000000001664: D6130006 24160902
	v_fmac_f32_e32 v4, v6, v3                                  // 00000000166C: 56080706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001670: BF870091
	v_fma_f32 v2, -v2, v4, v5                                  // 000000001674: D6130002 24160902
	v_div_fmas_f32 v2, v2, v3, v4                              // 00000000167C: D6370002 04120702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001684: BF870091
	v_div_fixup_f32 v1, v2, v1, 1.0                            // 000000001688: D6270001 03CA0302
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v1                     // 000000001690: 7C2802FF 0F800000
	v_cndmask_b32_e64 v2, 1.0, 0x4f800000, vcc_lo              // 000000001698: D5010002 01A9FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A4: BF870091
	v_mul_f32_e32 v1, v1, v2                                   // 0000000016A8: 10020501
	v_sqrt_f32_e32 v2, v1                                      // 0000000016AC: 7E046701
	s_waitcnt_depctr 0xfff                                     // 0000000016B0: BF880FFF
	v_add_nc_u32_e32 v3, -1, v2                                // 0000000016B4: 4A0604C1
	v_add_nc_u32_e32 v4, 1, v2                                 // 0000000016B8: 4A080481
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016BC: BF870112
	v_fma_f32 v5, -v3, v2, v1                                  // 0000000016C0: D6130005 24060503
	v_fma_f32 v6, -v4, v2, v1                                  // 0000000016C8: D6130006 24060504
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016D0: BF870092
	v_cmp_ge_f32_e64 s0, 0, v5                                 // 0000000016D4: D4160000 00020A80
	v_cndmask_b32_e64 v2, v2, v3, s0                           // 0000000016DC: D5010002 00020702
	v_cndmask_b32_e64 v3, 1.0, 0x37800000, vcc_lo              // 0000000016E4: D5010003 01A9FEF2 37800000
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000016F0: BF8704A4
	v_cmp_lt_f32_e32 vcc_lo, 0, v6                             // 0000000016F4: 7C220C80
	s_lshl_b64 s[0:1], s[2:3], 2                               // 0000000016F8: 84808202
	s_add_u32 s0, s4, s0                                       // 0000000016FC: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001700: 82010105
	v_cndmask_b32_e32 v2, v2, v4, vcc_lo                       // 000000001704: 02040902
	v_cmp_class_f32_e64 vcc_lo, v1, 0x260                      // 000000001708: D47E006A 0001FF01 00000260
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001714: BF870092
	v_mul_f32_e32 v2, v3, v2                                   // 000000001718: 10040503
	v_cndmask_b32_e32 v1, v2, v1, vcc_lo                       // 00000000171C: 02020302
	global_store_b32 v0, v1, s[0:1]                            // 000000001720: DC6A0000 00000100
	s_nop 0                                                    // 000000001728: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000172C: BFB60003
	s_endpgm                                                   // 000000001730: BFB00000
