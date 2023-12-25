
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n59>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_mov_b32_e32 v0, 0                                        // 000000001610: 7E000280
	s_lshl_b64 s[6:7], s[4:5], 1                               // 000000001614: 84868104
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s6                                       // 00000000161C: 80020602
	s_addc_u32 s3, s3, s7                                      // 000000001620: 82030703
	global_load_u16 v1, v0, s[2:3]                             // 000000001624: DC4A0000 01020000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 00000000162C: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001630: BF8700C9
	s_add_u32 s0, s0, s2                                       // 000000001634: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001638: 82010301
	s_waitcnt vmcnt(0)                                         // 00000000163C: BF8903F7
	v_cvt_f32_u32_e32 v1, v1                                   // 000000001640: 7E020D01
	v_mul_f32_e32 v1, 0xbfb8aa3b, v1                           // 000000001644: 100202FF BFB8AA3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 00000000164C: BF870131
	v_cmp_gt_f32_e32 vcc_lo, 0xc2fc0000, v1                    // 000000001650: 7C2802FF C2FC0000
	v_cndmask_b32_e64 v3, 0, 0x42800000, vcc_lo                // 000000001658: D5010003 01A9FE80 42800000
	v_cndmask_b32_e64 v2, 1.0, 0x1f800000, vcc_lo              // 000000001664: D5010002 01A9FEF2 1F800000
	v_add_f32_e32 v1, v1, v3                                   // 000000001670: 06020701
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001674: BF8700B1
	v_exp_f32_e32 v1, v1                                       // 000000001678: 7E024B01
	s_waitcnt_depctr 0xfff                                     // 00000000167C: BF880FFF
	v_mul_f32_e32 v1, v2, v1                                   // 000000001680: 10020302
	v_add_f32_e32 v1, 1.0, v1                                  // 000000001684: 060202F2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001688: BF870121
	v_div_scale_f32 v2, null, v1, v1, 1.0                      // 00000000168C: D6FC7C02 03CA0301
	v_div_scale_f32 v5, vcc_lo, 1.0, v1, 1.0                   // 000000001694: D6FC6A05 03CA02F2
	v_rcp_f32_e32 v3, v2                                       // 00000000169C: 7E065502
	s_waitcnt_depctr 0xfff                                     // 0000000016A0: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 0000000016A4: D6130004 23CA0702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016AC: BF870091
	v_fmac_f32_e32 v3, v4, v3                                  // 0000000016B0: 56060704
	v_mul_f32_e32 v4, v5, v3                                   // 0000000016B4: 10080705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016B8: BF870091
	v_fma_f32 v6, -v2, v4, v5                                  // 0000000016BC: D6130006 24160902
	v_fmac_f32_e32 v4, v6, v3                                  // 0000000016C4: 56080706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016C8: BF870091
	v_fma_f32 v2, -v2, v4, v5                                  // 0000000016CC: D6130002 24160902
	v_div_fmas_f32 v2, v2, v3, v4                              // 0000000016D4: D6370002 04120702
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016DC: BF870001
	v_div_fixup_f32 v1, v2, v1, 1.0                            // 0000000016E0: D6270001 03CA0302
	global_store_b32 v0, v1, s[0:1]                            // 0000000016E8: DC6A0000 00000100
	s_nop 0                                                    // 0000000016F0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016F4: BFB60003
	s_endpgm                                                   // 0000000016F8: BFB00000
