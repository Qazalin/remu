
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_3n75>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001600: F4080100 F8000000
	s_mov_b32 s2, s15                                          // 000000001608: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 00000000160C: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001614: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s0, s6, s2                                       // 00000000161C: 80000206
	s_addc_u32 s1, s7, s3                                      // 000000001620: 82010307
	s_load_b32 s0, s[0:1], null                                // 000000001624: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000162C: BF89FC07
	v_cvt_f32_u32_e32 v0, s0                                   // 000000001630: 7E000C00
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 000000001634: BF870121
	v_div_scale_f32 v1, null, v0, v0, 1.0                      // 000000001638: D6FC7C01 03CA0100
	v_div_scale_f32 v4, vcc_lo, 1.0, v0, 1.0                   // 000000001640: D6FC6A04 03CA00F2
	v_rcp_f32_e32 v2, v1                                       // 000000001648: 7E045501
	s_waitcnt_depctr 0xfff                                     // 00000000164C: BF880FFF
	v_fma_f32 v3, -v1, v2, 1.0                                 // 000000001650: D6130003 23CA0501
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001658: BF870091
	v_fmac_f32_e32 v2, v3, v2                                  // 00000000165C: 56040503
	v_mul_f32_e32 v3, v4, v2                                   // 000000001660: 10060504
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001664: BF870091
	v_fma_f32 v5, -v1, v3, v4                                  // 000000001668: D6130005 24120701
	v_fmac_f32_e32 v3, v5, v2                                  // 000000001670: 56060505
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001674: BF870091
	v_fma_f32 v1, -v1, v3, v4                                  // 000000001678: D6130001 24120701
	v_div_fmas_f32 v1, v1, v2, v3                              // 000000001680: D6370001 040E0501
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001688: BF870091
	v_div_fixup_f32 v0, v1, v0, 1.0                            // 00000000168C: D6270000 03CA0101
	v_cmp_gt_f32_e32 vcc_lo, 0xf800000, v0                     // 000000001694: 7C2800FF 0F800000
	v_cndmask_b32_e64 v1, 1.0, 0x4f800000, vcc_lo              // 00000000169C: D5010001 01A9FEF2 4F800000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_mul_f32_e32 v0, v0, v1                                   // 0000000016AC: 10000300
	v_sqrt_f32_e32 v1, v0                                      // 0000000016B0: 7E026700
	s_waitcnt_depctr 0xfff                                     // 0000000016B4: BF880FFF
	v_add_nc_u32_e32 v2, -1, v1                                // 0000000016B8: 4A0402C1
	v_add_nc_u32_e32 v3, 1, v1                                 // 0000000016BC: 4A060281
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)// 0000000016C0: BF870112
	v_fma_f32 v4, -v2, v1, v0                                  // 0000000016C4: D6130004 24020302
	v_fma_f32 v5, -v3, v1, v0                                  // 0000000016CC: D6130005 24020303
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016D4: BF870092
	v_cmp_ge_f32_e64 s0, 0, v4                                 // 0000000016D8: D4160000 00020880
	v_cndmask_b32_e64 v1, v1, v2, s0                           // 0000000016E0: D5010001 00020501
	v_cndmask_b32_e64 v2, 1.0, 0x37800000, vcc_lo              // 0000000016E8: D5010002 01A9FEF2 37800000
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_2)// 0000000016F4: BF870154
	v_cmp_lt_f32_e32 vcc_lo, 0, v5                             // 0000000016F8: 7C220A80
	s_add_u32 s0, s4, s2                                       // 0000000016FC: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001700: 82010305
	v_cndmask_b32_e32 v1, v1, v3, vcc_lo                       // 000000001704: 02020701
	v_cmp_class_f32_e64 vcc_lo, v0, 0x260                      // 000000001708: D47E006A 0001FF00 00000260
	v_mul_f32_e32 v1, v2, v1                                   // 000000001714: 10020302
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001718: BF870001
	v_dual_cndmask_b32 v0, v1, v0 :: v_dual_mov_b32 v1, 0      // 00000000171C: CA500101 00000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001724: DC6A0000 00000001
	s_nop 0                                                    // 00000000172C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001730: BFB60003
	s_endpgm                                                   // 000000001734: BFB00000
