
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n9>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_mul_f32_e64 v0, 0x3fb8aa3b, s2                           // 000000001638: D5080000 000004FF 3FB8AA3B
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)// 000000001644: BF870131
	v_cmp_gt_f32_e32 vcc_lo, 0xc2fc0000, v0                    // 000000001648: 7C2800FF C2FC0000
	v_cndmask_b32_e64 v2, 0, 0x42800000, vcc_lo                // 000000001650: D5010002 01A9FE80 42800000
	v_cndmask_b32_e64 v1, 1.0, 0x1f800000, vcc_lo              // 00000000165C: D5010001 01A9FEF2 1F800000
	v_add_f32_e32 v0, v0, v2                                   // 000000001668: 06000500
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000166C: BF8700C1
	v_exp_f32_e32 v0, v0                                       // 000000001670: 7E004B00
	s_waitcnt_depctr 0xfff                                     // 000000001674: BF880FFF
	v_mul_f32_e32 v0, v1, v0                                   // 000000001678: 10000101
	v_max_f32_e64 v1, s2, s2                                   // 00000000167C: D5100001 00000402
	v_dual_sub_f32 v0, 1.0, v0 :: v_dual_max_f32 v1, 0, v1     // 000000001684: C95400F2 00000280
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000168C: BF870091
	v_max_f32_e32 v0, 0, v0                                    // 000000001690: 20000080
	v_dual_sub_f32 v0, v1, v0 :: v_dual_mov_b32 v1, 0          // 000000001694: C9500101 00000080
	global_store_b32 v1, v0, s[0:1]                            // 00000000169C: DC6A0000 00000001
	s_nop 0                                                    // 0000000016A4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016A8: BFB60003
	s_endpgm                                                   // 0000000016AC: BFB00000
