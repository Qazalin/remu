
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n9>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0                                        // 000000001608: 7E000280
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_mov_b32 s4, s15                                          // 000000001610: BE84000F
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	s_add_u32 s2, s2, s15                                      // 000000001618: 80020F02
	s_addc_u32 s3, s3, s5                                      // 00000000161C: 82030503
	global_load_i8 v1, v0, s[2:3]                              // 000000001620: DC460000 01020000
	s_lshl_b64 s[2:3], s[4:5], 2                               // 000000001628: 84828204
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000162C: BF8700C9
	s_add_u32 s0, s0, s2                                       // 000000001630: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001634: 82010301
	s_waitcnt vmcnt(0)                                         // 000000001638: BF8903F7
	v_cvt_f32_i32_e32 v1, v1                                   // 00000000163C: 7E020B01
	v_mul_f32_e32 v1, 0xbfb8aa3b, v1                           // 000000001640: 100202FF BFB8AA3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001648: BF870131
	v_cmp_gt_f32_e32 vcc_lo, 0xc2fc0000, v1                    // 00000000164C: 7C2802FF C2FC0000
	v_cndmask_b32_e64 v3, 0, 0x42800000, vcc_lo                // 000000001654: D5010003 01A9FE80 42800000
	v_cndmask_b32_e64 v2, 1.0, 0x1f800000, vcc_lo              // 000000001660: D5010002 01A9FEF2 1F800000
	v_add_f32_e32 v1, v1, v3                                   // 00000000166C: 06020701
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001670: BF8700B1
	v_exp_f32_e32 v1, v1                                       // 000000001674: 7E024B01
	s_waitcnt_depctr 0xfff                                     // 000000001678: BF880FFF
	v_mul_f32_e32 v1, v2, v1                                   // 00000000167C: 10020302
	v_add_f32_e32 v1, 1.0, v1                                  // 000000001680: 060202F2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001684: BF870121
	v_div_scale_f32 v2, null, v1, v1, 1.0                      // 000000001688: D6FC7C02 03CA0301
	v_div_scale_f32 v5, vcc_lo, 1.0, v1, 1.0                   // 000000001690: D6FC6A05 03CA02F2
	v_rcp_f32_e32 v3, v2                                       // 000000001698: 7E065502
	s_waitcnt_depctr 0xfff                                     // 00000000169C: BF880FFF
	v_fma_f32 v4, -v2, v3, 1.0                                 // 0000000016A0: D6130004 23CA0702
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_fmac_f32_e32 v3, v4, v3                                  // 0000000016AC: 56060704
	v_mul_f32_e32 v4, v5, v3                                   // 0000000016B0: 10080705
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016B4: BF870091
	v_fma_f32 v6, -v2, v4, v5                                  // 0000000016B8: D6130006 24160902
	v_fmac_f32_e32 v4, v6, v3                                  // 0000000016C0: 56080706
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016C4: BF870091
	v_fma_f32 v2, -v2, v4, v5                                  // 0000000016C8: D6130002 24160902
	v_div_fmas_f32 v2, v2, v3, v4                              // 0000000016D0: D6370002 04120702
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000016D8: BF870001
	v_div_fixup_f32 v1, v2, v1, 1.0                            // 0000000016DC: D6270001 03CA0302
	global_store_b32 v0, v1, s[0:1]                            // 0000000016E4: DC6A0000 00000100
	s_nop 0                                                    // 0000000016EC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016F0: BFB60003
	s_endpgm                                                   // 0000000016F4: BFB00000
